# Flake8 for Python 3.x incompatible use of print operator

Runs [hacking](https://github.com/openstack-dev/hacking) selecting only the error: `H233  Python 3.x incompatible use of print operator`

## Fixing H233 issues

The __2to3__ utility that ships with Python can automatically fix all print statements so that they are compatible with both Python 2 and Python 3.

At the root of your project, run `2to3 -f print .` to see all the lines that need to be changed and then run `2to3 -f print -w .` to actually write (-w) the changes into the files.

https://docs.python.org/2/library/2to3.html

---

__143__ instances of Python 3.x incompatible use of print operator found in https://github.com/QUVA-Lab/artemis

$ __pip install hacking__

$ __flake8 . --count --select=H233 --show-source --statistics__
```
./artemis/examples/demo_mnist_logreg.py:65:13: H233  Python 3.x incompatible use of print operator
            print 'Epoch {epoch}: Test Error: {test}%, Training Error: {train}%'.format(epoch=iteration_info.epoch, test=test_error, train=training_error)
            ^

./artemis/examples/demo_mnist_logreg.py:122:13: H233  Python 3.x incompatible use of print operator
            print ex.name.ljust(60) \
            ^

./artemis/experiments/demo_experiments.py:76:13: H233  Python 3.x incompatible use of print operator
            print 'Epoch {epoch}: Test Cost: {test}, Training Cost: {train}'.format(epoch=float(i)/n_training_samples, test=test_cost, train=training_cost)
            ^

./artemis/experiments/experiment_record.py:225:13: H233  Python 3.x incompatible use of print operator
            print 'Saving Result for Experiment "%s"' % (self.get_id(),)
            ^

./artemis/experiments/experiment_record.py:316:13: H233  Python 3.x incompatible use of print operator
            print error_text
            ^

./artemis/experiments/experiment_record_view.py:136:9: H233  Python 3.x incompatible use of print operator
        print deepstr(result)
        ^

./artemis/experiments/experiment_record_view.py:184:5: H233  Python 3.x incompatible use of print operator
    print tabulate(rows)
    ^

./artemis/experiments/experiment_record_view.py:201:9: H233  Python 3.x incompatible use of print operator
        print '... No records to show ...'
        ^

./artemis/experiments/experiment_record_view.py:212:13: H233  Python 3.x incompatible use of print operator
            print hang_notice
            ^

./artemis/experiments/experiment_record_view.py:228:9: H233  Python 3.x incompatible use of print operator
        print side_by_side(strings, max_linewidth=128)
        ^

./artemis/experiments/experiment_record_view.py:231:13: H233  Python 3.x incompatible use of print operator
            print string
            ^

./artemis/experiments/experiment_record_view.py:307:9: H233  Python 3.x incompatible use of print operator
        print tabulate.tabulate(rows, headers=headers, tablefmt='simple')
        ^

./artemis/experiments/test_experiment_record.py:40:9: H233  Python 3.x incompatible use of print operator
        print 'aaa'
        ^

./artemis/experiments/test_experiment_record.py:48:9: H233  Python 3.x incompatible use of print operator
        print 'bbb'
        ^

./artemis/experiments/test_experiment_record.py:112:9: H233  Python 3.x incompatible use of print operator
        print get_experiment_info('experiment_test_function')
        ^

./artemis/experiments/test_experiment_record.py:126:13: H233  Python 3.x incompatible use of print operator
            print '123'
            ^

./artemis/experiments/test_experiment_record.py:127:13: H233  Python 3.x incompatible use of print operator
            print 'abc'
            ^

./artemis/experiments/test_experiment_record.py:148:5: H233  Python 3.x incompatible use of print operator
    print c
    ^

./artemis/experiments/test_experiment_record.py:164:9: H233  Python 3.x incompatible use of print operator
        print c
        ^

./artemis/experiments/test_experiment_record.py:200:5: H233  Python 3.x incompatible use of print operator
    print 'aaa'
    ^

./artemis/experiments/test_experiment_record.py:298:9: H233  Python 3.x incompatible use of print operator
        print "^^^ Dont't worry, the above is not actually an error, we were just asserting that we caught the error."
        ^

./artemis/experiments/test_experiment_record_view_and_ui.py:10:5: H233  Python 3.x incompatible use of print operator
    print str(result) + 'aaa'
    ^

./artemis/experiments/test_experiment_record_view_and_ui.py:18:5: H233  Python 3.x incompatible use of print operator
    print ', '.join('{}: {}'.format(k, results[k]) for k in sorted(results.keys()))
    ^

./artemis/experiments/test_experiment_record_view_and_ui.py:27:5: H233  Python 3.x incompatible use of print operator
    print 'xxx' if a==1 else 'yyy'
    ^

./artemis/experiments/test_experiment_record_view_and_ui.py:63:9: H233  Python 3.x incompatible use of print operator
        print '='*100+'\n ARGTABLE \n'+'='*100
        ^

./artemis/experiments/test_experiment_record_view_and_ui.py:66:9: H233  Python 3.x incompatible use of print operator
        print '='*100+'\n SHOW \n'+'='*100
        ^

./artemis/experiments/test_experiment_record_view_and_ui.py:171:13: H233  Python 3.x incompatible use of print operator
            print 'xxxxx'
            ^

./artemis/experiments/test_experiment_record_view_and_ui.py:172:13: H233  Python 3.x incompatible use of print operator
            print 'yyyyy'
            ^

./artemis/experiments/test_matplotlib_dependency_is_soft.py:18:9: H233  Python 3.x incompatible use of print operator
        print 'aaa'
        ^

./artemis/experiments/ui.py:28:9: H233  Python 3.x incompatible use of print operator
        print message
        ^

./artemis/experiments/ui.py:186:13: H233  Python 3.x incompatible use of print operator
            print "==================== Experiments ===================="
            ^

./artemis/experiments/ui.py:189:13: H233  Python 3.x incompatible use of print operator
            print self.get_experiment_list_str(self.exp_record_dict, just_last_record=self.just_last_record,
            ^

./artemis/experiments/ui.py:193:17: H233  Python 3.x incompatible use of print operator
                print '[Filtered with "{}" to show {}/{} experiments]'.format(self._filter, len(self.exp_record_dict), len(all_experiments))
                ^

./artemis/experiments/ui.py:194:13: H233  Python 3.x incompatible use of print operator
            print '-----------------------------------------------------'
            ^

./artemis/experiments/ui.py:317:13: H233  Python 3.x incompatible use of print operator
            print '\n\n... Close all figures to return to experiment browser ...'
            ^

./artemis/experiments/ui.py:340:21: H233  Python 3.x incompatible use of print operator
                    print record.get_error_trace()
                    ^

./artemis/experiments/ui.py:345:9: H233  Python 3.x incompatible use of print operator
        print '{} out of {} Records will be deleted.'.format(len(records), sum(len(recs) for recs in self.exp_record_dict.values()))
        ^

./artemis/experiments/ui.py:347:13: H233  Python 3.x incompatible use of print operator
            print ExperimentRecordBrowser.get_record_table(records, )
            ^

./artemis/experiments/ui.py:351:13: H233  Python 3.x incompatible use of print operator
            print 'Records deleted.'
            ^

./artemis/experiments/ui.py:363:13: H233  Python 3.x incompatible use of print operator
            print self.get_experiment_list_str(exps_to_records, just_last_record=self.just_last_record, view_mode=self.view_mode, raise_display_errors=self.raise_display_errors)
            ^

./artemis/experiments/ui.py:370:13: H233  Python 3.x incompatible use of print operator
            print ExperimentRecordBrowser.get_record_table(records)
            ^

./artemis/experiments/ui.py:375:9: H233  Python 3.x incompatible use of print operator
        print side_by_side([get_record_full_string(rec) for rec in records], max_linewidth=128)
        ^

./artemis/experiments/ui.py:391:9: H233  Python 3.x incompatible use of print operator
        print output
        ^

./artemis/experiments/ui.py:409:9: H233  Python 3.x incompatible use of print operator
        print "\n".join([surround+k+surround for k in self.exp_record_dict.keys()])
        ^

./artemis/experiments/ui.py:511:13: H233  Python 3.x incompatible use of print operator
            print "=============== Experiment Records =================="
            ^

./artemis/experiments/ui.py:513:13: H233  Python 3.x incompatible use of print operator
            print self.get_record_table([load_experiment_record(rid) for rid in self.record_ids], raise_display_errors = self.raise_display_errors)
            ^

./artemis/experiments/ui.py:514:13: H233  Python 3.x incompatible use of print operator
            print '-----------------------------------------------------'
            ^

./artemis/experiments/ui.py:517:17: H233  Python 3.x incompatible use of print operator
                print 'Not showing all experiments.  Type "showall" to see all experiments, or "viewfilters" to view current filters.'
                ^

./artemis/experiments/ui.py:521:17: H233  Python 3.x incompatible use of print operator
                print "You need to specify an command.  Press h for help."
                ^

./artemis/experiments/ui.py:557:9: H233  Python 3.x incompatible use of print operator
        print self.get_record_table(self._select_records(user_range), headers=['Identifier', 'Args'])
        ^

./artemis/experiments/ui.py:577:9: H233  Python 3.x incompatible use of print operator
        print 'Found the following Records: '
        ^

./artemis/experiments/ui.py:578:9: H233  Python 3.x incompatible use of print operator
        print self.get_record_table([rid for rid in self.record_ids if filter_text in rid])
        ^

./artemis/experiments/ui.py:583:9: H233  Python 3.x incompatible use of print operator
        print 'We will delete the following experiments:'
        ^

./artemis/experiments/ui.py:584:9: H233  Python 3.x incompatible use of print operator
        print self.get_record_table(ids)
        ^

./artemis/fileman/config_files.py:134:5: H233  Python 3.x incompatible use of print operator
    print 'Contents of {}:\n-------------\n'.format(config_path)
    ^

./artemis/fileman/config_files.py:136:9: H233  Python 3.x incompatible use of print operator
        print f.read()
        ^

./artemis/fileman/disk_memoize.py:173:5: H233  Python 3.x incompatible use of print operator
    print 'Removed %s memos.' % (len(all_memos))
    ^

./artemis/fileman/file_getter.py:31:9: H233  Python 3.x incompatible use of print operator
        print 'Downloading file from url: "%s"...' % (url, )
        ^

./artemis/fileman/file_getter.py:34:9: H233  Python 3.x incompatible use of print operator
        print '...Done.'
        ^

./artemis/fileman/file_getter.py:37:13: H233  Python 3.x incompatible use of print operator
            print 'Processing downloaded data...'
            ^

./artemis/fileman/file_getter.py:96:9: H233  Python 3.x incompatible use of print operator
        print 'Downloading archive from url: "%s"...' % (url, )
        ^

./artemis/fileman/file_getter.py:98:9: H233  Python 3.x incompatible use of print operator
        print '...Done.'
        ^

./artemis/fileman/images2gif.py:571:13: H233  Python 3.x incompatible use of print operator
            print 'Saved GIF at: %s' % (self.filename, )
            ^

./artemis/fileman/persistent_print.py:64:5: H233  Python 3.x incompatible use of print operator
    print read_print()
    ^

./artemis/fileman/smart_io.py:36:9: H233  Python 3.x incompatible use of print operator
        print 'Saved object <%s at %s> to file: "%s"' % (obj.__class__.__name__, hex(id(object)), local_path)
        ^

./artemis/fileman/test_persistent_print.py:14:5: H233  Python 3.x incompatible use of print operator
    print 'ddd'
    ^

./artemis/fileman/test_persistent_print.py:16:9: H233  Python 3.x incompatible use of print operator
        print 'fff'
        ^

./artemis/fileman/test_persistent_print.py:17:9: H233  Python 3.x incompatible use of print operator
        print 'ggg'
        ^

./artemis/fileman/test_persistent_print.py:18:5: H233  Python 3.x incompatible use of print operator
    print 'hhh'
    ^

./artemis/fileman/test_persistent_print.py:26:9: H233  Python 3.x incompatible use of print operator
        print 'fff'
        ^

./artemis/fileman/test_persistent_print.py:27:9: H233  Python 3.x incompatible use of print operator
        print 'ggg'
        ^

./artemis/fileman/test_persistent_print.py:28:5: H233  Python 3.x incompatible use of print operator
    print 'hhh'
    ^

./artemis/fileman/test_persistent_print.py:41:5: H233  Python 3.x incompatible use of print operator
    print 'aaa'
    ^

./artemis/fileman/test_persistent_print.py:42:5: H233  Python 3.x incompatible use of print operator
    print 'bbb'
    ^

./artemis/fileman/test_persistent_print.py:48:5: H233  Python 3.x incompatible use of print operator
    print 'ccc'
    ^

./artemis/fileman/test_persistent_print.py:49:5: H233  Python 3.x incompatible use of print operator
    print 'ddd'
    ^

./artemis/fileman/test_persistent_print.py:59:5: H233  Python 3.x incompatible use of print operator
    print 'eee'
    ^

./artemis/fileman/test_persistent_print.py:60:5: H233  Python 3.x incompatible use of print operator
    print 'fff'
    ^

./artemis/general/call_timer.py:20:13: H233  Python 3.x incompatible use of print operator
            print 'Timer {}: {} iterations in {:.3}s ({:.3} iterations/second)'.format(self.name, self.call_count, current_time-self.start_time, self.call_count/(current_time-self.start_time))
            ^

./artemis/general/display.py:180:13: H233  Python 3.x incompatible use of print operator
            print self.block_header
            ^

./artemis/general/display.py:199:13: H233  Python 3.x incompatible use of print operator
            print '```'
            ^

./artemis/general/ezprofile.py:41:13: H233  Python 3.x incompatible use of print operator
            print '{} Started'.format(self.profiler_name)
            ^

./artemis/general/ezprofile.py:53:9: H233  Python 3.x incompatible use of print operator
        print self.get_report()
        ^

./artemis/general/mymath.py:340:9: H233  Python 3.x incompatible use of print operator
        print 'sdfdsf: {}'.format(np.max(np.abs(x)))
        ^

./artemis/general/profile.py:27:9: H233  Python 3.x incompatible use of print operator
        print surround_with_header('Profile for "{}"'.format(command), width=100, char='=')
        ^

./artemis/general/profile.py:29:9: H233  Python 3.x incompatible use of print operator
        print '='*100
        ^

./artemis/general/progress_indicator.py:47:13: H233  Python 3.x incompatible use of print operator
            print 'Progress%s: %s%%.  %.1fs Elapsed, %.1fs Remaining.%s' \
            ^

./artemis/general/report_collector.py:13:13: H233  Python 3.x incompatible use of print operator
            print update
            ^

./artemis/general/report_collector.py:22:9: H233  Python 3.x incompatible use of print operator
        print '='*15+' Report '+'='*15+'\n'+self.get_report_text()+'\n'+'='*40+'\n'
        ^

./artemis/general/test_display.py:24:9: H233  Python 3.x incompatible use of print operator
        print 'aaa'
        ^

./artemis/general/test_display.py:25:9: H233  Python 3.x incompatible use of print operator
        print 'bbb'
        ^

./artemis/general/test_display.py:27:13: H233  Python 3.x incompatible use of print operator
            print 'ccc'
            ^

./artemis/general/test_display.py:28:13: H233  Python 3.x incompatible use of print operator
            print 'ddd'
            ^

./artemis/general/test_display.py:30:17: H233  Python 3.x incompatible use of print operator
                print 'eee'
                ^

./artemis/general/test_display.py:31:17: H233  Python 3.x incompatible use of print operator
                print 'fff'
                ^

./artemis/general/test_display.py:32:13: H233  Python 3.x incompatible use of print operator
            print 'ggg'
            ^

./artemis/general/test_display.py:33:13: H233  Python 3.x incompatible use of print operator
            print 'hhh'
            ^

./artemis/general/test_display.py:34:9: H233  Python 3.x incompatible use of print operator
        print 'iii'
        ^

./artemis/general/test_display.py:35:9: H233  Python 3.x incompatible use of print operator
        print 'jjj'
        ^

./artemis/general/test_display.py:60:5: H233  Python 3.x incompatible use of print operator
    print 'String 1:\n{}'.format(str1)
    ^

./artemis/general/test_display.py:61:5: H233  Python 3.x incompatible use of print operator
    print 'String 2:\n{}'.format(str2)
    ^

./artemis/general/test_display.py:64:5: H233  Python 3.x incompatible use of print operator
    print 'Side by side:\n{}'.format(out)
    ^

./artemis/general/test_display.py:93:5: H233  Python 3.x incompatible use of print operator
    print string_desc
    ^

./artemis/general/test_display.py:126:9: H233  Python 3.x incompatible use of print operator
        print 'a'
        ^

./artemis/general/test_display.py:128:13: H233  Python 3.x incompatible use of print operator
            print 'b'
            ^

./artemis/general/test_display.py:129:9: H233  Python 3.x incompatible use of print operator
        print 'c'
        ^

./artemis/general/test_functional.py:23:9: H233  Python 3.x incompatible use of print operator
        print infer_arg_values(func, 1, 2, 5)
        ^

./artemis/general/test_functional.py:41:9: H233  Python 3.x incompatible use of print operator
        print infer_arg_values(func_with_defaults, 1, 2, 5)
        ^

./artemis/general/test_functional.py:56:9: H233  Python 3.x incompatible use of print operator
        print infer_arg_values(func_with_kwargs, 1, 2, 5)
        ^

./artemis/general/test_mymath.py:50:9: H233  Python 3.x incompatible use of print operator
        print 'Error for %s: %.4f True, %.4f Sample.' % (method, true_error, sample_error)
        ^

./artemis/general/test_tables.py:21:9: H233  Python 3.x incompatible use of print operator
        print tabulate.tabulate(rows)
        ^

./artemis/ml/datasets/books.py:63:5: H233  Python 3.x incompatible use of print operator
    print read_book(book, n_characters)
    ^

./artemis/ml/datasets/ilsvrc.py:36:5: H233  Python 3.x incompatible use of print operator
    print 'Loading %s' % (identifier, )
    ^

./artemis/ml/datasets/ilsvrc.py:43:5: H233  Python 3.x incompatible use of print operator
    print 'Done.'
    ^

./artemis/ml/datasets/imagenet.py:21:5: H233  Python 3.x incompatible use of print operator
    print 'Loading %s image URLs....' % (n_images, )
    ^

./artemis/ml/datasets/imagenet.py:27:5: H233  Python 3.x incompatible use of print operator
    print 'Done.'
    ^

./artemis/ml/datasets/imagenet.py:57:5: H233  Python 3.x incompatible use of print operator
    print ixs
    ^

./artemis/ml/datasets/newsgroups.py:146:9: H233  Python 3.x incompatible use of print operator
        print '%s: %s' % (targets, inputs)
        ^

./artemis/ml/predictors/predictor_comparison.py:70:9: H233  Python 3.x incompatible use of print operator
        print '%s\nRunning predictor %s\n%s' % ('='*20, predictor_name, '-'*20)
        ^

./artemis/ml/predictors/predictor_comparison.py:94:5: H233  Python 3.x incompatible use of print operator
    print 'Done!'
    ^

./artemis/ml/predictors/predictor_comparison.py:147:9: H233  Python 3.x incompatible use of print operator
        print 'Scores: %s' % (scores, )
        ^

./artemis/ml/predictors/predictor_comparison.py:192:13: H233  Python 3.x incompatible use of print operator
            print 'Scores at Epoch %s: %s, after %.2fs' % (current_epoch, ', '.join('%s: %.3f' % (set_name, score) for set_name, score in scores), time.time()-start_time)
            ^

./artemis/ml/predictors/train_and_test.py:33:5: H233  Python 3.x incompatible use of print operator
    print 'Training Predictor %s...' % (predictor, )
    ^

./artemis/ml/predictors/train_and_test.py:36:5: H233  Python 3.x incompatible use of print operator
    print 'Done.'
    ^

./artemis/ml/predictors/train_and_test.py:502:9: H233  Python 3.x incompatible use of print operator
        print results.get_table()
        ^

./artemis/ml/predictors/train_and_test.py:513:9: H233  Python 3.x incompatible use of print operator
        print 'Epoch {} (after {:.3g}s)'.format(info.epoch, info.time)
        ^

./artemis/ml/predictors/train_and_test.py:524:5: H233  Python 3.x incompatible use of print operator
    print tabulate.tabulate(rows)
    ^

./artemis/ml/predictors/train_and_test.py:562:13: H233  Python 3.x incompatible use of print operator
            print 'Epoch {}.  Rate: {:.3g}s/epoch'.format(info.epoch, rate)
            ^

./artemis/ml/predictors/train_and_test.py:655:13: H233  Python 3.x incompatible use of print operator
            print 'Epoch {}: {} = {}'.format(epoch, self.print_variable_name, new_value)
            ^

./artemis/ml/tools/test_iteration.py:75:13: H233  Python 3.x incompatible use of print operator
            print info.done
            ^

./artemis/plotting/bokeh_backend.py:92:5: H233  Python 3.x incompatible use of print operator
    print _GRIDPLOT
    ^

./artemis/plotting/bokeh_backend.py:204:13: H233  Python 3.x incompatible use of print operator
            print self._plots.data_source.data
            ^

./artemis/plotting/demo_dbplot.py:47:13: H233  Python 3.x incompatible use of print operator
            print 'Frame Rate: {:3g}FPS'.format(1./(time.time() - t_start))
            ^

./artemis/plotting/demo_dbplot.py:55:9: H233  Python 3.x incompatible use of print operator
        print 'aaa'
        ^

./artemis/plotting/demo_dbplot.py:58:9: H233  Python 3.x incompatible use of print operator
        print 'bbb'
        ^

./artemis/plotting/fast.py:4:5: H233  Python 3.x incompatible use of print operator
    print "Cannot Import scipy weave.  That's ok for now, you won't be able to use the fastplot function."
    ^

./artemis/plotting/test_live_plotting.py:34:13: H233  Python 3.x incompatible use of print operator
            print 'Average Frame Rate: %.2f FPS' % (i/(time.time()-start_time), )
            ^

./artemis/plotting/test_live_plotting.py:56:13: H233  Python 3.x incompatible use of print operator
            print 'Average Frame Rate: %.2f FPS' % (i/(time.time()-start_time), )
            ^

./artemis/remote/test_child_processes.py:110:5: H233  Python 3.x incompatible use of print operator
    print 'a+b={}'.format(a+b)
    ^

./artemis/remote/test_child_processes.py:125:5: H233  Python 3.x incompatible use of print operator
    print 'hello hello hello'
    ^

./artemis/remote/test_virtualenv.py:32:5: H233  Python 3.x incompatible use of print operator
    print packages
    ^

./artemis/remote/virtualenv.py:62:13: H233  Python 3.x incompatible use of print operator
            print err
            ^

./artemis/remote/virtualenv.py:106:13: H233  Python 3.x incompatible use of print operator
            print "The following locally installed packages are missing in the virtualenv at %s:" % ip_address
            ^

./artemis/remote/virtualenv.py:107:13: H233  Python 3.x incompatible use of print operator
            print '\n'.join(['%s: %s (%s)' % (i, key, missing_packages[key]) for i, key in enumerate(missing_packages.keys())])
            ^

./artemis/remote/virtualenv.py:144:13: H233  Python 3.x incompatible use of print operator
            print "The following locally installed packages are installed in a different version than on the virtualenv at %s:" % ip_address
            ^

./artemis/remote/virtualenv.py:145:13: H233  Python 3.x incompatible use of print operator
            print '\n'.join(['%s: %s (local: %s) => (remote: %s)' % (i, key, different_versions[key][0], different_versions[key][1]) for i, key in enumerate(different_versions.keys())])
            ^

146     H233  Python 3.x incompatible use of print operator
