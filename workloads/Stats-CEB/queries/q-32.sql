SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE u.Id = v.UserId
  AND v.UserId = ph.UserId
  AND ph.UserId = c.UserId
  AND c.CreationDate >= CAST('2010-08-12 20:33:46' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-13 19:26:55' AS timestamp)
  AND ph.CreationDate >= CAST('2011-04-11 14:46:09' AS timestamp)
  AND ph.CreationDate <= CAST('2014-08-17 16:37:23' AS timestamp)
  AND v.CreationDate >= CAST('2010-07-26 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-12 00:00:00' AS timestamp)
  AND u.Views >= 0
  AND u.Views <= 783
  AND u.DownVotes >= 0
  AND u.DownVotes <= 1
  AND u.UpVotes <= 123;
