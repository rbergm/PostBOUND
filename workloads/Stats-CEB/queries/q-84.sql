SELECT COUNT(*)
FROM comments AS c, votes AS v, users AS u
WHERE u.Id = c.UserId
  AND u.Id = v.UserId
  AND c.CreationDate >= CAST('2010-07-27 15:46:34' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-12 08:15:14' AS timestamp)
  AND v.CreationDate >= CAST('2010-07-19 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-10 00:00:00' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-03 01:06:41' AS timestamp);
